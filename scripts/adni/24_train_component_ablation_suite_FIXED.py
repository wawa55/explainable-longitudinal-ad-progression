from pathlib import Path
import copy
import json
import math
import random
import time
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import nibabel as nib

# ==============================================================================================================
# ADNI — STEP 24: FIVE-SEED COMPONENT ABLATION TRAINING
# ==============================================================================================================
# Trains ONLY:
#   A1_NO_TIME_ENCODING
#   A2_NO_TRANSFORMER
#   A3_LATEST_VISIT_ONLY
#
# Safety:
#   * TRAIN + VALIDATION only
#   * TEST dataframe / TEST DataLoader are never created
#   * Existing M0 checkpoints are never modified
#   * Resume-safe: completed checkpoints are reused
#   * Same frozen M0 training protocol for every ablation
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "06_splits"
MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"
SCRIPT_DIR = ROOT / "scripts"
DATA_FILE = SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
MODEL_SOURCE = SCRIPT_DIR / "exact_m0_model_v3.py"
LOCK_FILE = RESULT_DIR / "ablation_v1" / "ADNI_COMPONENT_ABLATION_LOCK_V1.json"
OUT_DIR = RESULT_DIR / "ablation_v1"
CKPT_DIR = MODEL_DIR / "ablation_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [2026, 3407, 5813, 7291, 9109]
ABLATIONS = ["A1_NO_TIME_ENCODING", "A2_NO_TRANSFORMER", "A3_LATEST_VISIT_ONLY"]

BATCH_SIZE = 2
MAX_EPOCHS = 50
PATIENCE = 10
MIN_DELTA = 1e-4
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
EXPECTED_PARAMS_FULL = 4_269_109
SOURCE_SPATIAL_SHAPE = (197, 233, 189)
MODEL_SPATIAL_SHAPE = (160, 192, 160)
DAYS_PER_YEAR = 365.25
ADMIN_CUTOFF_DAYS = 5.0 * DAYS_PER_YEAR
INTERVAL_EDGES_DAYS = np.asarray(
    [0.0, 1.0*DAYS_PER_YEAR, 2.0*DAYS_PER_YEAR, 3.0*DAYS_PER_YEAR, 4.0*DAYS_PER_YEAR, 5.0*DAYS_PER_YEAR],
    dtype=np.float64,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def find_col(df, candidates, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise RuntimeError("Could not find any of columns: " + ", ".join(candidates))
    return None

def parse_list_field(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    try:
        import json as _json
        x = _json.loads(s)
        if isinstance(x, list):
            return x
    except Exception:
        pass
    try:
        import ast
        x = ast.literal_eval(s)
        if isinstance(x, (list, tuple)):
            return list(x)
    except Exception:
        pass
    raise ValueError(f"Cannot parse list field: {s[:300]}")

def center_crop_3d(volume, target_shape=MODEL_SPATIAL_SHAPE):
    src = np.asarray(volume.shape)
    tgt = np.asarray(target_shape)
    if np.any(src < tgt):
        raise RuntimeError(f"Source shape {tuple(src)} smaller than target {tuple(tgt)}")
    start = (src - tgt) // 2
    end = start + tgt
    cropped = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    if cropped.shape != tuple(target_shape):
        raise RuntimeError(f"Crop mismatch: {cropped.shape} != {tuple(target_shape)}")
    return cropped

def load_mri_array(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI missing: {path}")
    nii = nib.load(str(path))
    volume = np.asarray(nii.dataobj, dtype=np.float32)
    if volume.shape != SOURCE_SPATIAL_SHAPE:
        raise RuntimeError(f"Unexpected MRI shape {volume.shape}; expected {SOURCE_SPATIAL_SHAPE}")
    if not np.isfinite(volume).all():
        raise RuntimeError(f"Non-finite MRI values: {path}")
    return center_crop_3d(volume)

def make_discrete_survival_target(event, time_days):
    event = int(event)
    time_days = float(time_days)
    y = np.zeros(5, dtype=np.float32)
    mask = np.zeros(5, dtype=np.float32)
    observed_time = min(time_days, ADMIN_CUTOFF_DAYS)
    event_within_60m = event == 1 and time_days <= ADMIN_CUTOFF_DAYS

    if event_within_60m:
        event_interval = int(np.searchsorted(INTERVAL_EDGES_DAYS[1:], time_days, side="left"))
        event_interval = min(max(event_interval, 0), 4)
        if event_interval > 0:
            mask[:event_interval] = 1.0
            y[:event_interval] = 0.0
        mask[event_interval] = 1.0
        y[event_interval] = 1.0
    else:
        for j in range(5):
            if observed_time >= INTERVAL_EDGES_DAYS[j+1]:
                mask[j] = 1.0
                y[j] = 0.0
    return y, mask

class LandmarkDataset(Dataset):
    def __init__(self, df, latest_only=False):
        self.df = df.reset_index(drop=True).copy()
        self.latest_only = bool(latest_only)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = str(row["subject_id"])
        image_paths = parse_list_field(row["image_paths"])
        days_from_first = np.asarray(parse_list_field(row["days_from_first_mri"]), dtype=np.float32)
        days_before_landmark = np.asarray(parse_list_field(row["days_before_landmark"]), dtype=np.float32)

        if not (len(image_paths) == len(days_from_first) == len(days_before_landmark)):
            raise RuntimeError(f"Sequence-length mismatch at row {idx}")
        if len(image_paths) == 0:
            raise RuntimeError(f"No MRI paths at row {idx}")

        if self.latest_only:
            image_paths = [image_paths[-1]]
            days_from_first = np.asarray([days_from_first[-1]], dtype=np.float32)
            days_before_landmark = np.asarray([days_before_landmark[-1]], dtype=np.float32)

        images = []
        for path in image_paths:
            arr = load_mri_array(path)
            images.append(torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0))
        images = torch.stack(images, dim=0)

        y, mask = make_discrete_survival_target(
            row["event"],
            row["time_to_event_or_censor_days"],
        )

        return {
            "subject_id": subject_id,
            "images": images,
            "years_from_first": torch.tensor(days_from_first / DAYS_PER_YEAR, dtype=torch.float32),
            "years_before_landmark": torch.tensor(days_before_landmark / DAYS_PER_YEAR, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
            "target_mask": torch.tensor(mask, dtype=torch.float32),
        }

def collate_longitudinal(batch):
    b = len(batch)
    max_t = max(x["images"].shape[0] for x in batch)
    c, d, h, w = batch[0]["images"].shape[1:]
    images = torch.zeros((b, max_t, c, d, h, w), dtype=torch.float32)
    visit_mask = torch.zeros((b, max_t), dtype=torch.bool)
    yff = torch.zeros((b, max_t), dtype=torch.float32)
    ybl = torch.zeros((b, max_t), dtype=torch.float32)

    for i, x in enumerate(batch):
        t = x["images"].shape[0]
        images[i, :t] = x["images"]
        visit_mask[i, :t] = True
        yff[i, :t] = x["years_from_first"]
        ybl[i, :t] = x["years_before_landmark"]

    return {
        "subject_id": [x["subject_id"] for x in batch],
        "images": images,
        "visit_mask": visit_mask,
        "years_from_first": yff,
        "years_before_landmark": ybl,
        "target": torch.stack([x["target"] for x in batch]),
        "target_mask": torch.stack([x["target_mask"] for x in batch]),
    }

class InformativeSubjectBalancedSampler(Sampler):
    def __init__(self, dataframe, target_masks, seed):
        self.df = dataframe.reset_index(drop=True).copy()
        self.seed = int(seed)
        informative = np.asarray(
            [float(np.asarray(m).sum()) > 0.0 for m in target_masks],
            dtype=bool,
        )
        self.subject_to_indices = {}
        for idx, row in self.df.iterrows():
            if not informative[idx]:
                continue
            sid = str(row["subject_id"])
            self.subject_to_indices.setdefault(sid, []).append(idx)
        self.subjects = sorted(self.subject_to_indices.keys())
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        subjects = self.subjects.copy()
        rng.shuffle(subjects)
        indices = []
        for sid in subjects:
            candidates = self.subject_to_indices[sid]
            position = self.epoch % len(candidates)
            indices.append(candidates[position])
        return iter(indices)

    def __len__(self):
        return len(self.subjects)

class DiscreteTimeSurvivalNLL(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target, target_mask):
        element = self.bce(logits, target)
        observed = (element * target_mask).sum(dim=1)
        informative = target_mask.sum(dim=1) > 0
        if not bool(informative.any()):
            return logits.sum() * 0.0
        return observed[informative].mean()

def import_m0():
    spec = importlib.util.spec_from_file_location("exact_m0_model_v3", MODEL_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class A1NoTimeEncoding(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.mri_encoder = copy.deepcopy(base.mri_encoder)
        self.transformer = copy.deepcopy(base.transformer)
        self.final_norm = copy.deepcopy(base.final_norm)
        self.survival_head = copy.deepcopy(base.survival_head)

    def forward(self, images, visit_mask, years_from_first, years_before_landmark):
        b, t = images.shape[:2]
        flat = images.reshape(b*t, *images.shape[2:])
        fm = visit_mask.reshape(-1)
        emb = torch.zeros((b*t, 256), device=images.device, dtype=images.dtype)
        if bool(fm.any()):
            emb[fm] = self.mri_encoder(flat[fm])
        x = emb.reshape(b, t, 256)
        x = self.transformer(x, src_key_padding_mask=~visit_mask)
        mf = visit_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mf).sum(1) / mf.sum(1).clamp_min(1.0)
        return self.survival_head(self.final_norm(pooled))

class A2NoTransformer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.mri_encoder = copy.deepcopy(base.mri_encoder)
        self.time_encoder = copy.deepcopy(base.time_encoder)
        self.final_norm = copy.deepcopy(base.final_norm)
        self.survival_head = copy.deepcopy(base.survival_head)

    def forward(self, images, visit_mask, years_from_first, years_before_landmark):
        b, t = images.shape[:2]
        flat = images.reshape(b*t, *images.shape[2:])
        fm = visit_mask.reshape(-1)
        emb = torch.zeros((b*t, 256), device=images.device, dtype=images.dtype)
        if bool(fm.any()):
            emb[fm] = self.mri_encoder(flat[fm])
        emb = emb.reshape(b, t, 256)
        x = emb + self.time_encoder(years_from_first, years_before_landmark)
        mf = visit_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mf).sum(1) / mf.sum(1).clamp_min(1.0)
        return self.survival_head(self.final_norm(pooled))

def make_model(ablation, mod):
    base = mod.ExactM0LongitudinalSurvivalModel_V3()
    if ablation == "A1_NO_TIME_ENCODING":
        return A1NoTimeEncoding(base)
    if ablation == "A2_NO_TRANSFORMER":
        return A2NoTransformer(base)
    if ablation == "A3_LATEST_VISIT_ONLY":
        return base
    raise ValueError(ablation)

def run_model(model, batch):
    return model(
        images=batch["images"].to(DEVICE, non_blocking=True),
        visit_mask=batch["visit_mask"].to(DEVICE, non_blocking=True),
        years_from_first=batch["years_from_first"].to(DEVICE, non_blocking=True),
        years_before_landmark=batch["years_before_landmark"].to(DEVICE, non_blocking=True),
    )

@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        target = batch["target"].to(DEVICE)
        mask = batch["target_mask"].to(DEVICE)
        logits = run_model(model, batch)
        loss = loss_fn(logits, target, mask)
        bs = target.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)

def train_one(ablation, seed, train_df, val_df, mod, train_target_masks):
    ckpt_path = CKPT_DIR / f"{ablation}_SEED_{seed}_BEST_V1.pt"
    history_path = OUT_DIR / f"{ablation}_SEED_{seed}_training_history.csv"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("completed", False):
            print(f"REUSE COMPLETE: {ckpt_path.name} | epoch={ckpt.get('epoch')} | val_NLL={ckpt.get('validation_loss')}")
            return {
                "ablation": ablation, "seed": seed, "status": "REUSED",
                "best_epoch": ckpt.get("epoch"), "validation_nll": ckpt.get("validation_loss"),
                "checkpoint": str(ckpt_path)
            }

    set_seed(seed)
    latest_only = (ablation == "A3_LATEST_VISIT_ONLY")
    train_ds = LandmarkDataset(train_df, latest_only=latest_only)
    val_ds = LandmarkDataset(val_df, latest_only=latest_only)

    sampler = InformativeSubjectBalancedSampler(train_df, train_target_masks, seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=collate_longitudinal
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=collate_longitudinal
    )

    model = make_model(ablation, mod).to(DEVICE)
    loss_fn = DiscreteTimeSurvivalNLL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("\n" + "="*110)
    print(f"{ablation} — SEED {seed}")
    print("="*110)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train informative subjects/epoch: {len(sampler)}")
    print(f"Train batches/epoch: {len(train_loader)}")
    print(f"Validation landmarks: {len(val_ds)}")
    print("TEST Dataset/DataLoader: NOT CREATED")

    best_val = math.inf
    best_epoch = -1
    bad_epochs = 0
    history = []
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        sampler.set_epoch(epoch - 1)
        model.train()
        train_total, train_n = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(DEVICE)
            mask = batch["target_mask"].to(DEVICE)
            logits = run_model(model, batch)
            loss = loss_fn(logits, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            bs = target.shape[0]
            train_total += float(loss.item()) * bs
            train_n += bs

        train_loss = train_total / max(train_n, 1)
        val_loss = validate(model, val_loader, loss_fn)
        improved = val_loss < (best_val - MIN_DELTA)

        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "experiment": "ADNI_COMPONENT_ABLATION_V1",
                "ablation": ablation,
                "seed": seed,
                "epoch": epoch,
                "validation_loss": float(val_loss),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_protocol": {
                    "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                    "min_delta": MIN_DELTA, "lr": LR, "weight_decay": WEIGHT_DECAY,
                    "gradient_clip": GRAD_CLIP
                },
                "test_used": False,
                "completed": False,
            }, ckpt_path)
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch, "train_nll": train_loss, "validation_nll": val_loss,
            "best_validation_nll": best_val, "bad_epochs": bad_epochs
        })
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | train_NLL={train_loss:.6f} "
            f"| val_NLL={val_loss:.6f} | best={best_val:.6f} @ {best_epoch:02d} "
            f"| patience={bad_epochs}/{PATIENCE}"
        )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["completed"] = True
    ckpt["elapsed_minutes"] = (time.time() - start) / 60.0
    torch.save(ckpt, ckpt_path)

    return {
        "ablation": ablation, "seed": seed, "status": "TRAINED",
        "best_epoch": int(best_epoch), "validation_nll": float(best_val),
        "checkpoint": str(ckpt_path)
    }

def main():
    print("="*110)
    print("STEP 24 — FIVE-SEED COMPONENT ABLATION TRAINING")
    print("="*110)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    for p in [DATA_FILE, MODEL_SOURCE, LOCK_FILE]:
        print(f"{p.name:60s} | {'FOUND' if p.exists() else 'MISSING'}")
        if not p.exists():
            raise RuntimeError(f"Missing required file: {p}")

    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    if lock.get("experiment") != "ADNI_COMPONENT_ABLATION_V1":
        raise RuntimeError("Unexpected ablation lock.")
    if lock.get("safety", {}).get("test_inference_performed") is not False:
        raise RuntimeError("Safety lock mismatch.")

    # Load only enough CSV information to explicitly filter TRAIN and VALIDATION.
    all_df = pd.read_csv(DATA_FILE)
    split_col = find_col(all_df, ["split", "partition", "set"])
    subject_col = find_col(all_df, ["subject_id", "participant_id", "PTID"])

    train_df = all_df[all_df[split_col].astype(str).str.upper().eq("TRAIN")].copy()
    val_df = all_df[all_df[split_col].astype(str).str.upper().eq("VALIDATION")].copy()

    # Delete the full dataframe reference immediately; no TEST dataframe is constructed.
    del all_df

    print("\nFROZEN DEVELOPMENT DATA")
    print("-"*110)
    print(f"TRAIN landmarks: {len(train_df)}")
    print(f"TRAIN participants: {train_df[subject_col].nunique()}")
    print(f"VALIDATION landmarks: {len(val_df)}")
    print(f"VALIDATION participants: {val_df[subject_col].nunique()}")
    print("TEST dataframe: NOT CREATED")
    print("TEST DataLoader: NOT CREATED")
    print("TEST inference: WILL NOT BE PERFORMED")

    required_columns = [
        "split", "subject_id", "landmark_id", "event",
        "time_to_event_or_censor_days", "image_paths",
        "days_from_first_mri", "days_before_landmark",
    ]
    missing_columns = [c for c in required_columns if c not in train_df.columns]
    if missing_columns:
        raise RuntimeError("Frozen dataset schema mismatch. Missing: " + ", ".join(missing_columns))
    print("Frozen dataset schema: PASS")

    if len(train_df) != 2530 or train_df[subject_col].nunique() != 680:
        raise RuntimeError("Frozen TRAIN counts do not match.")
    if len(val_df) != 521 or val_df[subject_col].nunique() != 146:
        raise RuntimeError("Frozen VALIDATION counts do not match.")

    train_target_masks = []
    for _, row in train_df.iterrows():
        _, mask = make_discrete_survival_target(
            row["event"],
            row["time_to_event_or_censor_days"],
        )
        train_target_masks.append(mask)

    informative_subjects = train_df.loc[
        np.asarray([np.asarray(m).sum() > 0 for m in train_target_masks], dtype=bool),
        "subject_id",
    ].astype(str).nunique()
    print(f"Informative TRAIN participants: {informative_subjects}")
    if informative_subjects != 574:
        raise RuntimeError(f"Expected 574 informative TRAIN participants, got {informative_subjects}")

    mod = import_m0()
    full = mod.ExactM0LongitudinalSurvivalModel_V3()
    if sum(p.numel() for p in full.parameters()) != EXPECTED_PARAMS_FULL:
        raise RuntimeError("Exact M0 parameter audit failed.")

    results = []
    for ablation in ABLATIONS:
        for seed in SEEDS:
            results.append(train_one(ablation, seed, train_df, val_df, mod, train_target_masks))
            pd.DataFrame(results).to_csv(OUT_DIR / "ADNI_COMPONENT_ABLATION_TRAINING_SUMMARY_V1.csv", index=False)

    summary = pd.DataFrame(results)
    print("\n" + "="*110)
    print("ABLATION TRAINING COMPLETE")
    print("="*110)
    print(summary.to_string(index=False))

    means = summary.groupby("ablation")["validation_nll"].agg(["mean", "std", "min", "max"])
    means.to_csv(OUT_DIR / "ADNI_COMPONENT_ABLATION_VALIDATION_SUMMARY_V1.csv")
    print("\nVALIDATION NLL ACROSS FIVE SEEDS")
    print("-"*110)
    print(means)

    report = OUT_DIR / "ADNI_COMPONENT_ABLATION_TRAINING_REPORT_V1.txt"
    report.write_text(
        "ADNI COMPONENT ABLATION TRAINING V1\n"
        + "="*80 + "\n"
        + summary.to_string(index=False)
        + "\n\nVALIDATION NLL ACROSS FIVE SEEDS\n"
        + means.to_string()
        + "\n\nTEST DATAFRAME CREATED: NO"
        + "\nTEST DATALOADER CREATED: NO"
        + "\nTEST INFERENCE PERFORMED: NO"
        + "\nSTATUS: PASS — ALL REQUIRED ABLATION TRAINING RUNS COMPLETE\n",
        encoding="utf-8"
    )

    print("\nSaved:")
    print(" ", OUT_DIR / "ADNI_COMPONENT_ABLATION_TRAINING_SUMMARY_V1.csv")
    print(" ", OUT_DIR / "ADNI_COMPONENT_ABLATION_VALIDATION_SUMMARY_V1.csv")
    print(" ", report)
    print("\nSTATUS: PASS — ALL REQUIRED ABLATION TRAINING RUNS COMPLETE")
    print("TEST inference performed: NO")

if __name__ == "__main__":
    main()
