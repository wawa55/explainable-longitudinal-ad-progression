from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd


# ==============================================================================================================
# ADNI — FROZEN BENCHMARK COMPARISON SUITE LOCK
#
# PURPOSE
# -------
# Freeze the comparator models and evaluation rules BEFORE benchmark training.
#
# This prevents post-hoc benchmark selection after seeing TEST performance.
#
# Frozen benchmark suite:
#
#   B0 — Landmark-only 3D CNN survival model
#        Uses only the latest MRI available at the landmark.
#
#   B1 — Longitudinal 3D CNN + masked mean pooling survival model
#        Uses all MRI visits but NO temporal sequence model.
#
#   B2 — Longitudinal 3D CNN + GRU survival model
#        Uses all MRI visits and temporal order, but NO Transformer.
#
#   M0 — Frozen longitudinal 3D CNN + continuous-time Transformer survival model
#        Already trained/evaluated; this is the proposed model.
#
# Optional internal ablations M1/M2/M3 remain separate from external benchmark models.
#
# IMPORTANT
# ---------
# Same frozen subject split, same longitudinal landmark dataset, same 60-month discrete survival target,
# same TRAIN-only fitting rules, same validation-only model selection, and same final TEST metrics.
# ==============================================================================================================


ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "06_splits"
RESULT_DIR = ROOT / "08_results"

SUBJECT_SPLITS = SPLIT_DIR / "ADNI_FROZEN_SUBJECT_SPLITS_V1.csv"
LONGITUDINAL_DATASET = SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
SURVIVAL_TARGET_SPEC = SPLIT_DIR / "ADNI_FROZEN_SURVIVAL_TARGET_SPEC_60M_V2.json"
PRETEST_LOCK = RESULT_DIR / "ADNI_FINAL_PRETEST_ANALYSIS_LOCK_V1.json"
FINAL_RESULTS_LOCK = RESULT_DIR / "ADNI_FINAL_RESULTS_LOCK_V1.json"

OUTPUT_JSON = RESULT_DIR / "ADNI_BENCHMARK_SUITE_LOCK_V1.json"
OUTPUT_TXT = RESULT_DIR / "ADNI_BENCHMARK_SUITE_LOCK_V1.txt"
OUTPUT_HASHES = RESULT_DIR / "ADNI_BENCHMARK_SUITE_LOCK_HASHES_V1.csv"


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


BENCHMARK_SPEC = {
    "lock_version": "ADNI_BENCHMARK_SUITE_V1_2026-09-11",

    "status": "FROZEN BEFORE BENCHMARK TRAINING",

    "purpose": (
        "Fair architecture benchmark against the frozen M0 longitudinal CNN-Transformer "
        "using identical subject splits, survival targets, censoring rules, and evaluation metrics."
    ),

    "models": {
        "B0_landmark_only_3dcnn": {
            "name": "Landmark-only 3D CNN survival",
            "input": "latest MRI only",
            "temporal_model": "none",
            "time_encoding": "none",
            "mri_encoder": "same residual 3D CNN encoder family as M0",
            "survival_head": "same 5-interval discrete-time survival head family",
            "scientific_question": (
                "Does longitudinal history improve over a single latest MRI?"
            ),
        },

        "B1_longitudinal_mean_pool": {
            "name": "Longitudinal 3D CNN + masked mean pooling survival",
            "input": "all MRI visits",
            "temporal_model": "none",
            "time_encoding": "same continuous time encoder as M0",
            "aggregation": "masked mean of MRI+time embeddings",
            "mri_encoder": "same residual 3D CNN encoder family as M0",
            "survival_head": "same 5-interval discrete-time survival head family",
            "scientific_question": (
                "Does an explicit temporal sequence model improve over simple longitudinal aggregation?"
            ),
        },

        "B2_longitudinal_gru": {
            "name": "Longitudinal 3D CNN + GRU survival",
            "input": "all MRI visits",
            "temporal_model": "single-layer GRU",
            "time_encoding": "same continuous time encoder as M0",
            "hidden_dimension": 256,
            "mri_encoder": "same residual 3D CNN encoder family as M0",
            "survival_head": "same 5-interval discrete-time survival head family",
            "scientific_question": (
                "Does the Transformer outperform a conventional recurrent longitudinal model?"
            ),
        },

        "M0_proposed": {
            "name": "Longitudinal 3D CNN + continuous-time Transformer survival",
            "status": "already frozen",
            "parameters": 4269109,
            "temporal_model": "3-layer pre-norm TransformerEncoder",
            "d_model": 256,
            "heads": 8,
            "ffn_dimension": 512,
            "scientific_question": "proposed model",
        },
    },

    "common_training_rules": {
        "subject_split": "ADNI_SPLIT_V1_2026-09-07; no reshuffling",
        "training_subjects": 680,
        "validation_subjects": 146,
        "test_subjects": 146,
        "optimization_landmarks": (
            "TRAIN informative landmarks only through the same informative subject-balanced sampler"
        ),
        "validation_landmarks": "all frozen validation landmarks",
        "test_landmarks": "all frozen test landmarks",
        "survival_horizon_months": 60,
        "hazard_intervals_months": [
            [0, 12],
            [12, 24],
            [24, 36],
            [36, 48],
            [48, 60],
        ],
        "primary_horizons_months": [12, 24, 36],
        "secondary_horizons_months": [48, 60],
        "loss": (
            "same discrete-time survival NLL: sum observed interval NLL per informative landmark, "
            "then mean over informative landmarks"
        ),
        "model_selection": "validation NLL only",
        "test_model_selection": "prohibited",
        "test_recalibration": "prohibited",
        "augmentation": "none for primary benchmark suite",
        "mri_crop": [160, 192, 160],
        "source_preprocessing": "identical locked MRI preprocessing",
    },

    "optimization": {
        "seeds": [2026, 3407, 5813, 7291, 9109],
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "max_epochs": 30,
        "early_stopping_patience": 6,
        "batch_size": 2,
        "gradient_clip_norm": 1.0,
        "mixed_precision": False,
        "selection_metric": "validation NLL",
        "note": (
            "Same optimizer/training schedule across B0/B1/B2 to prioritize fairness. "
            "No benchmark-specific TEST tuning."
        ),
    },

    "evaluation": {
        "ensemble": "arithmetic mean of the five frozen seed cumulative risks for each benchmark",
        "primary_metrics": [
            "IPCW cumulative/dynamic AUC at 12m, 24m, 36m",
            "IPCW Brier score at 12m, 24m, 36m",
            "Uno C-index through 60m",
            "Harrell C-index through 60m",
            "Integrated Brier Score through 60m",
        ],
        "secondary_metrics": [
            "IPCW AUC at 48m and 60m",
            "IPCW Brier at 48m and 60m",
            "calibration intercept",
            "calibration slope",
            "ECE/MCE where defined",
        ],
        "uncertainty_intervals": (
            "2000 participant-cluster bootstrap percentile confidence intervals"
        ),
        "censoring_distribution": "TRAIN-only censoring KM",
        "unit_for_resampling": "participant",
        "comparison_rule": (
            "All benchmark models evaluated on identical frozen TEST landmarks and subjects."
        ),
    },

    "interpretation_rules": [
        "Benchmark comparison tests architecture value, not biological causality.",
        "B0 tests value of longitudinal history.",
        "B1 tests value of learned temporal sequence modeling beyond simple aggregation.",
        "B2 tests Transformer versus recurrent temporal modeling.",
        "M1/M2/M3 remain ablations and are not substitutes for B0/B1/B2.",
        "No benchmark may be added or removed based on TEST performance.",
    ],
}


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def require(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} missing:\n{path}")

    print(f"{label:42s}: PASS | {path}")


def main():
    print("=" * 120)
    print("ADNI BENCHMARK SUITE LOCK")
    print("=" * 120)

    print("Purpose: freeze fair benchmark models before benchmark training")
    print("TEST benchmark inference performed: NO")
    print()

    print("FILE AUDIT")
    print("-" * 120)

    for label, path in [
        ("Frozen subject splits", SUBJECT_SPLITS),
        ("Longitudinal dataset", LONGITUDINAL_DATASET),
        ("Survival target specification", SURVIVAL_TARGET_SPEC),
        ("Pre-TEST analysis lock", PRETEST_LOCK),
        ("Final M0 results lock", FINAL_RESULTS_LOCK),
    ]:
        require(path, label)

    split_df = pd.read_csv(SUBJECT_SPLITS)
    long_df = pd.read_csv(LONGITUDINAL_DATASET)

    print()
    print("FROZEN DATASET AUDIT")
    print("-" * 120)

    subjects = int(split_df["subject_id"].nunique())
    landmarks = int(len(long_df))

    split_counts = (
        split_df.groupby(
            split_df["split"].astype(str).str.upper()
        )["subject_id"]
        .nunique()
        .to_dict()
    )

    landmark_counts = (
        long_df.groupby(
            long_df["split"].astype(str).str.upper()
        )["landmark_id"]
        .count()
        .to_dict()
    )

    print(f"Subjects: {subjects}")
    print(f"Landmarks: {landmarks}")
    print(f"Subject split counts: {split_counts}")
    print(f"Landmark split counts: {landmark_counts}")

    if subjects != EXPECTED["subjects"]:
        raise RuntimeError("Subject count mismatch.")

    if landmarks != EXPECTED["landmarks"]:
        raise RuntimeError("Landmark count mismatch.")

    expected_subject_splits = {
        "TRAIN": EXPECTED["train_subjects"],
        "VALIDATION": EXPECTED["validation_subjects"],
        "TEST": EXPECTED["test_subjects"],
    }

    expected_landmark_splits = {
        "TRAIN": EXPECTED["train_landmarks"],
        "VALIDATION": EXPECTED["validation_landmarks"],
        "TEST": EXPECTED["test_landmarks"],
    }

    if split_counts != expected_subject_splits:
        raise RuntimeError(
            f"Subject split mismatch: {split_counts} != {expected_subject_splits}"
        )

    if landmark_counts != expected_landmark_splits:
        raise RuntimeError(
            f"Landmark split mismatch: {landmark_counts} != {expected_landmark_splits}"
        )

    print("Frozen dataset audit: PASS")

    lock = {
        **BENCHMARK_SPEC,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": {
            SUBJECT_SPLITS.name: sha256_file(SUBJECT_SPLITS),
            LONGITUDINAL_DATASET.name: sha256_file(LONGITUDINAL_DATASET),
            SURVIVAL_TARGET_SPEC.name: sha256_file(SURVIVAL_TARGET_SPEC),
            PRETEST_LOCK.name: sha256_file(PRETEST_LOCK),
            FINAL_RESULTS_LOCK.name: sha256_file(FINAL_RESULTS_LOCK),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)

    text = f"""====================================================================================================
ADNI BENCHMARK SUITE LOCK — V1
====================================================================================================

STATUS
------
Benchmark suite: FROZEN BEFORE BENCHMARK TRAINING
TEST benchmark inference: NOT YET RUN

MODELS
------
B0 — Landmark-only 3D CNN survival
     Question: Does longitudinal history improve over a single latest MRI?

B1 — Longitudinal 3D CNN + masked mean pooling survival
     Question: Does learned temporal sequence modeling improve over simple aggregation?

B2 — Longitudinal 3D CNN + GRU survival
     Question: Does the Transformer outperform a recurrent longitudinal model?

M0 — Longitudinal 3D CNN + continuous-time Transformer survival
     Status: ALREADY FROZEN PROPOSED MODEL

COMMON DATA
-----------
Subjects: 972
TRAIN / VALIDATION / TEST subjects: 680 / 146 / 146
Landmarks: 3634
TRAIN / VALIDATION / TEST landmarks: 2530 / 521 / 583
Subject reshuffling: PROHIBITED

COMMON SURVIVAL TARGET
----------------------
0–12, 12–24, 24–36, 36–48, 48–60 months
Primary horizons: 12 / 24 / 36 months
Secondary horizons: 48 / 60 months

COMMON OPTIMIZATION
-------------------
Seeds: 2026, 3407, 5813, 7291, 9109
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-4
Maximum epochs: 30
Early stopping patience: 6
Batch size: 2
Gradient clipping: 1.0
Selection metric: validation NLL only

COMMON EVALUATION
-----------------
Five-seed arithmetic-mean ensemble
IPCW AUC
IPCW Brier
Uno C-index
Harrell C-index
Integrated Brier Score
Calibration
2000 participant-cluster bootstrap confidence intervals
TRAIN-only censoring KM

POST-LOCK RULES
---------------
No benchmark may be selected based on TEST results.
No benchmark-specific TEST tuning.
No TEST recalibration.
No subject reshuffling.
M1/M2/M3 remain internal ablations.

NEXT PHASE
----------
Train B0, B1 and B2 under this frozen benchmark specification.

====================================================================================================
"""

    OUTPUT_TXT.write_text(text, encoding="utf-8")

    hash_rows = []

    for p in [
        SUBJECT_SPLITS,
        LONGITUDINAL_DATASET,
        SURVIVAL_TARGET_SPEC,
        PRETEST_LOCK,
        FINAL_RESULTS_LOCK,
        OUTPUT_JSON,
        OUTPUT_TXT,
    ]:
        hash_rows.append(
            {
                "artifact": p.name,
                "path": str(p),
                "sha256": sha256_file(p),
            }
        )

    with open(
        OUTPUT_HASHES,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact",
                "path",
                "sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(hash_rows)

    print()
    print("SAVED")
    print("-" * 120)
    print("JSON  :", OUTPUT_JSON)
    print("TXT   :", OUTPUT_TXT)
    print("HASHES:", OUTPUT_HASHES)

    print()
    print("=" * 120)
    print("BENCHMARK SUITE LOCK: COMPLETE")
    print("=" * 120)

    print("Models frozen: B0, B1, B2, M0")
    print("Five seeds: [2026, 3407, 5813, 7291, 9109]")
    print("Subject split: FROZEN")
    print("Survival target: FROZEN")
    print("Validation-only model selection: YES")
    print("TEST tuning: PROHIBITED")
    print("Next step: TRAIN B0/B1/B2")
    print("=" * 120)


if __name__ == "__main__":
    main()
