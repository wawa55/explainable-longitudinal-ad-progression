from pathlib import Path
import json, hashlib, importlib.util
import torch

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "06_splits"
MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"
SCRIPT_DIR = ROOT / "scripts"

DATA_FILE = SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
MODEL_SOURCE = SCRIPT_DIR / "exact_m0_model_v3.py"
OUT_DIR = RESULT_DIR / "ablation_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOCK_FILE = OUT_DIR / "ADNI_COMPONENT_ABLATION_LOCK_V1.json"
REPORT_FILE = OUT_DIR / "ADNI_COMPONENT_ABLATION_LOCK_V1.txt"

SEEDS = [2026, 3407, 5813, 7291, 9109]
REFERENCE_CHECKPOINTS = {
    2026: MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt",
    3407: MODEL_DIR / "ADNI_M0_SEED_3407_BEST_V1.pt",
    5813: MODEL_DIR / "ADNI_M0_SEED_5813_BEST_V1.pt",
    7291: MODEL_DIR / "ADNI_M0_SEED_7291_BEST_V1.pt",
    9109: MODEL_DIR / "ADNI_M0_SEED_9109_BEST_V1.pt",
}

TRAINING_SPEC = {
    "batch_size": 2,
    "max_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "early_stopping_patience": 10,
    "min_delta": 1e-4,
    "embedding_dim": 256,
    "transformer_layers": 3,
    "attention_heads": 8,
    "transformer_ff_dim": 512,
    "dropout": 0.10,
    "hazard_outputs": 5,
    "administrative_horizon_months": 60,
    "primary_horizons_months": [12, 24, 36],
    "secondary_horizons_months": [48, 60],
    "model_input_shape": [160, 192, 160],
    "train_sampler": "informative subject-balanced",
    "train_subjects_per_epoch_expected": 574,
    "validation_landmarks_expected": 521,
    "validation_subjects_expected": 146,
    "test_landmarks_expected": 583,
    "test_subjects_expected": 146,
}

ABLATIONS = {
    "A0_FULL_M0": {
        "status": "REFERENCE_EXISTING",
        "change": "none",
        "question": "Reference full longitudinal CNN-Transformer model."
    },
    "A1_NO_TIME_ENCODING": {
        "status": "TRAIN_REQUIRED",
        "change": "replace continuous-time embedding contribution by zeros",
        "question": "Does explicit irregular visit timing add predictive value?"
    },
    "A2_NO_TRANSFORMER": {
        "status": "TRAIN_REQUIRED",
        "change": "remove Transformer; directly masked-mean pool time-aware visit embeddings",
        "question": "Does cross-visit self-attention add value beyond time-aware pooling?"
    },
    "A3_LATEST_VISIT_ONLY": {
        "status": "TRAIN_REQUIRED",
        "change": "retain only latest MRI at each landmark while retaining its observed time covariates",
        "question": "Do earlier MRI examinations add incremental value beyond the latest MRI?"
    },
}

ANALYSIS_SPEC = {
    "selection": "best checkpoint within each seed by validation NLL only",
    "ensemble": "arithmetic mean of five seed cumulative-risk predictions",
    "comparisons": [
        "A1_NO_TIME_ENCODING vs A0_FULL_M0",
        "A2_NO_TRANSFORMER vs A0_FULL_M0",
        "A3_LATEST_VISIT_ONLY vs A0_FULL_M0",
    ],
    "metrics": [
        "AUC_12m","AUC_24m","AUC_36m",
        "Brier_12m","Brier_24m","Brier_36m",
        "Uno_C_60m","Harrell_C_60m","IBS_0_60m"
    ],
    "uncertainty": "participant-cluster bootstrap, 2000 same-resample replicates",
    "interpretation_rule": "Clear ablation effect only if paired 95% CI for favourable difference excludes zero; otherwise no clear ablation effect; do not claim equivalence.",
    "status": "POST_HOC_COMPONENT_ABLATION_LOCKED_BEFORE_ABLATION_OUTCOMES"
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    print("="*100)
    print("STEP 23 — ADNI COMPONENT ABLATION SUITE LOCK")
    print("="*100)

    required = [DATA_FILE, MODEL_SOURCE] + list(REFERENCE_CHECKPOINTS.values())
    for p in required:
        print(f"{p.name:55s} | {'FOUND' if p.exists() else 'MISSING'}")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required files:\n" + "\n".join(missing))

    spec = importlib.util.spec_from_file_location("exact_m0_model_v3", MODEL_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ExactM0LongitudinalSurvivalModel_V3()

    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nReference parameters: {nparams:,}")
    if nparams != 4_269_109:
        raise RuntimeError(f"Unexpected M0 parameter count: {nparams:,}")
    for attr in ["mri_encoder","time_encoder","transformer","final_norm","survival_head"]:
        if not hasattr(model, attr):
            raise RuntimeError(f"Missing expected M0 component: {attr}")

    audit = {}
    for seed, path in REFERENCE_CHECKPOINTS.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        audit[str(seed)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "checkpoint_seed": ckpt.get("seed"),
            "best_epoch": ckpt.get("epoch"),
            "validation_loss": ckpt.get("validation_loss"),
            "strict_load": True,
        }
        print(f"Seed {seed}: strict load PASS | epoch={ckpt.get('epoch')} | val_NLL={ckpt.get('validation_loss')}")

    lock = {
        "experiment": "ADNI_COMPONENT_ABLATION_V1",
        "reference_model": "M0_Longitudinal_MRI_Only_CNN_Transformer",
        "reference_model_class": "ExactM0LongitudinalSurvivalModel_V3",
        "reference_model_parameters": nparams,
        "reference_model_source": str(MODEL_SOURCE),
        "reference_model_source_sha256": sha256_file(MODEL_SOURCE),
        "dataset": str(DATA_FILE),
        "dataset_sha256": sha256_file(DATA_FILE),
        "seeds": SEEDS,
        "training_spec": TRAINING_SPEC,
        "ablations": ABLATIONS,
        "analysis_spec": ANALYSIS_SPEC,
        "reference_checkpoint_audit": audit,
        "safety": {
            "test_dataframe_loaded": False,
            "test_dataloader_created": False,
            "test_inference_performed": False,
            "training_performed": False,
            "existing_checkpoint_modified": False,
        },
    }

    LOCK_FILE.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    lines = [
        "="*100,
        "ADNI COMPONENT ABLATION SUITE — FROZEN LOCK V1",
        "="*100,
        f"Reference model: {lock['reference_model']}",
        f"Reference parameters: {nparams:,}",
        f"Seeds: {SEEDS}",
        "",
    ]
    for key, cfg in ABLATIONS.items():
        lines += [key, f"  status: {cfg['status']}", f"  change: {cfg['change']}", f"  question: {cfg['question']}", ""]
    lines += [
        "TEST DATA ACCESSED BY THIS SCRIPT: NO",
        "TRAINING PERFORMED BY THIS SCRIPT: NO",
        "STATUS: PASS — ABLATION DEFINITIONS LOCKED BEFORE ABLATION OUTCOMES",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\nLOCKED CONFIGURATIONS")
    print("-"*100)
    for key, cfg in ABLATIONS.items():
        print(f"{key:24s} | {cfg['status']:18s} | {cfg['question']}")
    print("\nSaved:")
    print(" ", LOCK_FILE)
    print(" ", REPORT_FILE)
    print("\nSTATUS: PASS — ABLATION DEFINITIONS LOCKED BEFORE ABLATION OUTCOMES")

if __name__ == "__main__":
    main()
